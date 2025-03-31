; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 497613

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child134 child303 child428 - child
    bread449 bread321 bread412 - bread-portion
    content410 content492 content487 - content-portion
    tray470 tray459 tray245 tray476 - tray
    table311 table24 table99 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray470 kitchen)
     (at tray459 kitchen)
     (at tray245 kitchen)
     (at tray476 kitchen)
     (at_kitchen_bread bread449)
     (at_kitchen_bread bread321)
     (at_kitchen_bread bread412)
     (at_kitchen_content content410)
     (at_kitchen_content content492)
     (at_kitchen_content content487)
     (not_allergic_gluten child134)
     (not_allergic_gluten child303)
     (not_allergic_gluten child428)
     (waiting child134 table311)
     (waiting child303 table311)
     (waiting child428 table99)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child134)
     (served child303)
     (served child428)
    )
  )
)
