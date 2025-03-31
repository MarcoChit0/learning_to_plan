; child-snack task with 2 children and 0.56 gluten factor 
; constant factor of 1.3
; random seed: 876054

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child426 child220 - child
    bread100 bread119 - bread-portion
    content353 content479 - content-portion
    tray107 tray356 - tray
    table188 table435 table213 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray107 kitchen)
     (at tray356 kitchen)
     (at_kitchen_bread bread100)
     (at_kitchen_bread bread119)
     (at_kitchen_content content353)
     (at_kitchen_content content479)
     (no_gluten_bread bread119)
     (no_gluten_content content353)
     (allergic_gluten child426)
     (not_allergic_gluten child220)
     (waiting child426 table213)
     (waiting child220 table188)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child426)
     (served child220)
    )
  )
)
