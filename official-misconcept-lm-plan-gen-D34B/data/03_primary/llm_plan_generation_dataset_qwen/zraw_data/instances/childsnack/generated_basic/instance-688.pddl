; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 102142

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child209 - child
    bread384 - bread-portion
    content247 - content-portion
    tray30 tray364 tray347 tray371 - tray
    table209 table185 table106 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray30 kitchen)
     (at tray364 kitchen)
     (at tray347 kitchen)
     (at tray371 kitchen)
     (at_kitchen_bread bread384)
     (at_kitchen_content content247)
     (not_allergic_gluten child209)
     (waiting child209 table209)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child209)
    )
  )
)
