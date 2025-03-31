; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 712453

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child352 child148 - child
    bread38 bread462 - bread-portion
    content72 content297 - content-portion
    tray420 tray264 - tray
    table107 table65 table394 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray420 kitchen)
     (at tray264 kitchen)
     (at_kitchen_bread bread38)
     (at_kitchen_bread bread462)
     (at_kitchen_content content72)
     (at_kitchen_content content297)
     (no_gluten_bread bread462)
     (no_gluten_content content297)
     (allergic_gluten child352)
     (not_allergic_gluten child148)
     (waiting child352 table107)
     (waiting child148 table65)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child352)
     (served child148)
    )
  )
)
