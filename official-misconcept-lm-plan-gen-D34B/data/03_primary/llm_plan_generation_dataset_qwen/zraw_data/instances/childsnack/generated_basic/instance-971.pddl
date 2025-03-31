; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 28543

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child434 child288 child477 - child
    bread393 bread407 bread7 - bread-portion
    content365 content58 content247 - content-portion
    tray33 tray73 tray104 tray16 - tray
    table353 table390 table411 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray33 kitchen)
     (at tray73 kitchen)
     (at tray104 kitchen)
     (at tray16 kitchen)
     (at_kitchen_bread bread393)
     (at_kitchen_bread bread407)
     (at_kitchen_bread bread7)
     (at_kitchen_content content365)
     (at_kitchen_content content58)
     (at_kitchen_content content247)
     (not_allergic_gluten child434)
     (not_allergic_gluten child288)
     (not_allergic_gluten child477)
     (waiting child434 table390)
     (waiting child288 table353)
     (waiting child477 table411)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child434)
     (served child288)
     (served child477)
    )
  )
)
