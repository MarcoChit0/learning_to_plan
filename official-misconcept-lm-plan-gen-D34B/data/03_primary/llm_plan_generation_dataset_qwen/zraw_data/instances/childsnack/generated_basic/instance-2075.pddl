; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 305733

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child440 - child
    bread271 - bread-portion
    content273 - content-portion
    tray212 tray217 tray380 tray413 - tray
    table381 table253 table408 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray212 kitchen)
     (at tray217 kitchen)
     (at tray380 kitchen)
     (at tray413 kitchen)
     (at_kitchen_bread bread271)
     (at_kitchen_content content273)
     (not_allergic_gluten child440)
     (waiting child440 table381)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child440)
    )
  )
)
