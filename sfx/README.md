# SFX

This directory contains the Audio and light sfx modules. Please refer below on how to use them.


## Factory

This is the object that creates all the below described elements. Essentially, reads a json file of the form:

        ```
        [
            "spell_name_1": [
                {"AudioEffect": "file/to/play"},
                {"LightEffect": [
                    {'group':'1', 'command': 'fade_up'},
                     ...
                    {'group':'1', 'command': 'brighten', 'payload': '50'},
                ]},
            ],
            ...
            "spell_name_N": [
                {"AudioEffect": "file/to/play"},
                {"LightEffect": [
                    {'group':'1', 'command': 'fade_up'},
                     ...
                    {'group':'1', 'command': 'brighten', 'payload': '50'},
                ]},
            ]

        ]
        ```
Where spell_name is the name you want the sequence of Effects to be run, such as 'alohomora' or 'arresto_momentum'. Please do realize that SpellNet should have a class named _exactly_ the same way for that spell to run, as the [controller](../core/controller) will run it using the name as the key.
By the same token, a dictionary with module names that implements all the EffectContainer method must be passed as a dictionary on the constructor, such as:

    ```
    {'AudioEffect': AudioEffect, 'LightEffect': LightEffect}
    ```
Realize how the keys for that dict matches the ones in the json file.



## Effect

An effect is an object that performs a part of the action of a spell, nominally, light and sound. It must inherit from the Effect class, implements a run() method and provide a unique name for reference.
Later on, these are grouped into the Effects container.

Example:

    ```
    an_effect = LightEffect(a_controller, a_light)
    an_effect.read_json("[{'group':'1', 'command': 'fade_up': 'payload': '50'}]")
    an_effect.run()
    ```

## Effect Container

This object is esentially alist of Effect objects that on the run command will run all of the available effects on sequence.

Example:

    ```
    from sfx.effect_container import EffectContainer
    alohomora = EffectContainer()
    alohomora.append(sound_for_alohomora)
    alohomora.append(light_for_alohomora)
    alohomora.run()
    ```

## Audio

This module essentially calls a mediaplayer (which *must* be installed beforehand) and runs the player for a given file.

Example:
    ```
    from sfx.audio.effect import AudioEffect
    audio_for_alohomora = AudioEffect('/path/to/alohomora/sfx/audio_file')
    audio_for_alohomora.run()
    ```
As this effect runs in a sepparate thread, it may be a good idea to use it in the first place, so the audio runs in background while other effects run.
I did not tested what happens if we try to run several audio effects simultanously, however, it is likely that an error will raise. Please do take a look at the config section, the PIPOTTER_EXTRA_AUDIO_COMMANDS segment for spectifics on your audio drivers


## Lights

This module does use the [MiLight python](https://github.com/McSwindler/python-milight) interface. Essentially, given a controller and a lightbulb spec, will run a sequence (that may be of a single) of commands.

Example:
    ```
    an_effect = LightEffect("[{'group':'1', 'command': 'fade_up': 'payload': '50'}]", a_controller, a_light)
    ```
The most important part here is the configuration json, that may take this form:

```json
        [
            {"group":"1", "command":"command", "payload": "value"},
            ...,
            {"group":"1", "command":"command", "payload": "value"}
        ]
```
* group is the integer of the group that will run the command.
* command is a valid MiLight python interface command. Check the link for details.
* payload if the value to be sent, and it is optional, as some commands, such as fade_out does not uses it.




