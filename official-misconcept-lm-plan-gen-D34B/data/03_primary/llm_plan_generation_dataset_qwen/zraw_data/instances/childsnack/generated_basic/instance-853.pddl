; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 145081

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child468 child375 - child
    bread386 bread2 - bread-portion
    content252 content468 - content-portion
    tray21 tray98 - tray
    table180 table486 table457 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray21 kitchen)
     (at tray98 kitchen)
     (at_kitchen_bread bread386)
     (at_kitchen_bread bread2)
     (at_kitchen_content content252)
     (at_kitchen_content content468)
     (no_gluten_bread bread2)
     (no_gluten_content content252)
     (allergic_gluten child375)
     (not_allergic_gluten child468)
     (waiting child468 table457)
     (waiting child375 table180)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child468)
     (served child375)
    )
  )
)
