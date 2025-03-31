; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 456239

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child359 - child
    bread345 - bread-portion
    content497 - content-portion
    tray246 tray271 tray58 tray171 - tray
    table254 table492 table101 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray246 kitchen)
     (at tray271 kitchen)
     (at tray58 kitchen)
     (at tray171 kitchen)
     (at_kitchen_bread bread345)
     (at_kitchen_content content497)
     (not_allergic_gluten child359)
     (waiting child359 table254)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child359)
    )
  )
)
