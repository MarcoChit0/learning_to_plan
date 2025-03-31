; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 821069

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child380 - child
    bread363 - bread-portion
    content364 - content-portion
    tray227 tray381 tray219 tray285 - tray
    table81 table199 table268 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray227 kitchen)
     (at tray381 kitchen)
     (at tray219 kitchen)
     (at tray285 kitchen)
     (at_kitchen_bread bread363)
     (at_kitchen_content content364)
     (not_allergic_gluten child380)
     (waiting child380 table268)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child380)
    )
  )
)
