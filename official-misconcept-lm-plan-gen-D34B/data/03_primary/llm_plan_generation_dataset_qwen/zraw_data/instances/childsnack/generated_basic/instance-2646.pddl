; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 419226

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child193 - child
    bread406 - bread-portion
    content131 - content-portion
    tray28 tray430 tray89 tray387 - tray
    table80 table398 table239 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray28 kitchen)
     (at tray430 kitchen)
     (at tray89 kitchen)
     (at tray387 kitchen)
     (at_kitchen_bread bread406)
     (at_kitchen_content content131)
     (not_allergic_gluten child193)
     (waiting child193 table239)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child193)
    )
  )
)
