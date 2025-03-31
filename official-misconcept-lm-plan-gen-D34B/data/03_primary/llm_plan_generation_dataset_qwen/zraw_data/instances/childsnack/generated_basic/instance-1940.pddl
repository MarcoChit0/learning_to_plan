; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 982821

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child81 - child
    bread199 - bread-portion
    content338 - content-portion
    tray129 tray350 tray399 - tray
    table345 table221 table292 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray129 kitchen)
     (at tray350 kitchen)
     (at tray399 kitchen)
     (at_kitchen_bread bread199)
     (at_kitchen_content content338)
     (not_allergic_gluten child81)
     (waiting child81 table292)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child81)
    )
  )
)
