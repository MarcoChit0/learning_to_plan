; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 644667

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child490 - child
    bread160 - bread-portion
    content424 - content-portion
    tray215 - tray
    table421 table280 table224 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray215 kitchen)
     (at_kitchen_bread bread160)
     (at_kitchen_content content424)
     (not_allergic_gluten child490)
     (waiting child490 table224)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child490)
    )
  )
)
