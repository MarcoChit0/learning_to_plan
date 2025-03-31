; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 484500

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child273 child432 - child
    bread294 bread451 - bread-portion
    content247 content102 - content-portion
    tray456 tray233 tray410 - tray
    table140 table181 table57 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray456 kitchen)
     (at tray233 kitchen)
     (at tray410 kitchen)
     (at_kitchen_bread bread294)
     (at_kitchen_bread bread451)
     (at_kitchen_content content247)
     (at_kitchen_content content102)
     (not_allergic_gluten child273)
     (not_allergic_gluten child432)
     (waiting child273 table140)
     (waiting child432 table57)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child273)
     (served child432)
    )
  )
)
