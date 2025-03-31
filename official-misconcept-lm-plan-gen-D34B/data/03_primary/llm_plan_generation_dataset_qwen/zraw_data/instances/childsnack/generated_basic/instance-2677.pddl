; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 404578

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child399 - child
    bread131 - bread-portion
    content118 - content-portion
    tray392 tray249 tray492 tray477 - tray
    table153 table65 table423 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray392 kitchen)
     (at tray249 kitchen)
     (at tray492 kitchen)
     (at tray477 kitchen)
     (at_kitchen_bread bread131)
     (at_kitchen_content content118)
     (not_allergic_gluten child399)
     (waiting child399 table65)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child399)
    )
  )
)
