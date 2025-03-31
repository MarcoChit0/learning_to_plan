; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 210853

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child8 - child
    bread143 - bread-portion
    content165 - content-portion
    tray274 tray179 - tray
    table249 table453 table248 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray274 kitchen)
     (at tray179 kitchen)
     (at_kitchen_bread bread143)
     (at_kitchen_content content165)
     (not_allergic_gluten child8)
     (waiting child8 table453)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child8)
    )
  )
)
