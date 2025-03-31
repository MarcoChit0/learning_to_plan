; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 897466

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child18 child483 - child
    bread21 bread199 - bread-portion
    content250 content60 - content-portion
    tray418 tray349 - tray
    table50 table156 table431 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray418 kitchen)
     (at tray349 kitchen)
     (at_kitchen_bread bread21)
     (at_kitchen_bread bread199)
     (at_kitchen_content content250)
     (at_kitchen_content content60)
     (not_allergic_gluten child18)
     (not_allergic_gluten child483)
     (waiting child18 table50)
     (waiting child483 table156)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child18)
     (served child483)
    )
  )
)
