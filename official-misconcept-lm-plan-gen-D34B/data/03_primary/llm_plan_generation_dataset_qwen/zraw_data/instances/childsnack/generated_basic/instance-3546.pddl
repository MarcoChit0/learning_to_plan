; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 687808

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child231 child420 - child
    bread245 bread56 - bread-portion
    content385 content258 - content-portion
    tray31 tray100 - tray
    table31 table307 table253 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray31 kitchen)
     (at tray100 kitchen)
     (at_kitchen_bread bread245)
     (at_kitchen_bread bread56)
     (at_kitchen_content content385)
     (at_kitchen_content content258)
     (not_allergic_gluten child420)
     (not_allergic_gluten child231)
     (waiting child231 table31)
     (waiting child420 table253)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child231)
     (served child420)
    )
  )
)
