; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 883225

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child257 child196 - child
    bread454 bread60 - bread-portion
    content119 content103 - content-portion
    tray413 tray251 tray194 - tray
    table170 table430 table466 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray413 kitchen)
     (at tray251 kitchen)
     (at tray194 kitchen)
     (at_kitchen_bread bread454)
     (at_kitchen_bread bread60)
     (at_kitchen_content content119)
     (at_kitchen_content content103)
     (not_allergic_gluten child257)
     (not_allergic_gluten child196)
     (waiting child257 table430)
     (waiting child196 table466)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child257)
     (served child196)
    )
  )
)
