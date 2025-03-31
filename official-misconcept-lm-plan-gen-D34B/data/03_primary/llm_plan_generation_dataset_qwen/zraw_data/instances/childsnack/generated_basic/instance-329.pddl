; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 509623

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child113 - child
    bread155 - bread-portion
    content357 - content-portion
    tray407 tray25 tray160 tray131 - tray
    table332 table137 table72 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray407 kitchen)
     (at tray25 kitchen)
     (at tray160 kitchen)
     (at tray131 kitchen)
     (at_kitchen_bread bread155)
     (at_kitchen_content content357)
     (not_allergic_gluten child113)
     (waiting child113 table137)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child113)
    )
  )
)
