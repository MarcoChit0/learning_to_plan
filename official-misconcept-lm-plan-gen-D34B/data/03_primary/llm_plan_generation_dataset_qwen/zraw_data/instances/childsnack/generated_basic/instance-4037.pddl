; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 454342

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child30 child393 child208 - child
    bread300 bread218 bread143 - bread-portion
    content279 content330 content163 - content-portion
    tray34 - tray
    table127 table93 table357 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray34 kitchen)
     (at_kitchen_bread bread300)
     (at_kitchen_bread bread218)
     (at_kitchen_bread bread143)
     (at_kitchen_content content279)
     (at_kitchen_content content330)
     (at_kitchen_content content163)
     (not_allergic_gluten child30)
     (not_allergic_gluten child393)
     (not_allergic_gluten child208)
     (waiting child30 table93)
     (waiting child393 table93)
     (waiting child208 table357)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child30)
     (served child393)
     (served child208)
    )
  )
)
