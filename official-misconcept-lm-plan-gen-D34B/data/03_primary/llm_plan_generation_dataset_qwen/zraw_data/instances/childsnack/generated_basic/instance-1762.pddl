; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 12673

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child327 - child
    bread158 - bread-portion
    content130 - content-portion
    tray269 - tray
    table421 table12 table457 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray269 kitchen)
     (at_kitchen_bread bread158)
     (at_kitchen_content content130)
     (not_allergic_gluten child327)
     (waiting child327 table421)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child327)
    )
  )
)
