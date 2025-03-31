; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 944125

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child414 child233 - child
    bread29 bread19 - bread-portion
    content101 content164 - content-portion
    tray154 - tray
    table147 table487 table125 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray154 kitchen)
     (at_kitchen_bread bread29)
     (at_kitchen_bread bread19)
     (at_kitchen_content content101)
     (at_kitchen_content content164)
     (no_gluten_bread bread29)
     (no_gluten_content content101)
     (allergic_gluten child414)
     (not_allergic_gluten child233)
     (waiting child414 table487)
     (waiting child233 table487)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child414)
     (served child233)
    )
  )
)
