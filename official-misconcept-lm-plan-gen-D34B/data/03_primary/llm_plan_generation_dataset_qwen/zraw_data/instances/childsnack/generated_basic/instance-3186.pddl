; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 568217

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child268 child110 - child
    bread443 bread133 - bread-portion
    content368 content415 - content-portion
    tray67 tray232 - tray
    table450 table433 table125 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray67 kitchen)
     (at tray232 kitchen)
     (at_kitchen_bread bread443)
     (at_kitchen_bread bread133)
     (at_kitchen_content content368)
     (at_kitchen_content content415)
     (not_allergic_gluten child268)
     (not_allergic_gluten child110)
     (waiting child268 table125)
     (waiting child110 table433)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child268)
     (served child110)
    )
  )
)
