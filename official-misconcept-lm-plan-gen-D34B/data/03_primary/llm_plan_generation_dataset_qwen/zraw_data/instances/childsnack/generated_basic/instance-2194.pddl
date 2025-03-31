; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 975223

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child200 - child
    bread102 - bread-portion
    content120 - content-portion
    tray51 - tray
    table63 table24 table163 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray51 kitchen)
     (at_kitchen_bread bread102)
     (at_kitchen_content content120)
     (not_allergic_gluten child200)
     (waiting child200 table63)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child200)
    )
  )
)
