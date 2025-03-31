; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 666325

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child121 - child
    bread355 - bread-portion
    content423 - content-portion
    tray446 tray338 tray125 tray212 - tray
    table365 table369 table345 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray446 kitchen)
     (at tray338 kitchen)
     (at tray125 kitchen)
     (at tray212 kitchen)
     (at_kitchen_bread bread355)
     (at_kitchen_content content423)
     (not_allergic_gluten child121)
     (waiting child121 table345)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child121)
    )
  )
)
