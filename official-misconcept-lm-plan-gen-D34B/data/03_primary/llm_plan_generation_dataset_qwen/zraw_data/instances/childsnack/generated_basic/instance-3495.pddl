; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 783086

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child403 child60 - child
    bread389 bread262 - bread-portion
    content427 content483 - content-portion
    tray23 tray483 - tray
    table364 table319 table61 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray23 kitchen)
     (at tray483 kitchen)
     (at_kitchen_bread bread389)
     (at_kitchen_bread bread262)
     (at_kitchen_content content427)
     (at_kitchen_content content483)
     (no_gluten_bread bread262)
     (no_gluten_content content483)
     (allergic_gluten child403)
     (not_allergic_gluten child60)
     (waiting child403 table61)
     (waiting child60 table61)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child403)
     (served child60)
    )
  )
)
