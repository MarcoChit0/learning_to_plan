; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 761416

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child236 - child
    bread40 - bread-portion
    content18 - content-portion
    tray267 tray318 tray388 - tray
    table276 table400 table390 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray267 kitchen)
     (at tray318 kitchen)
     (at tray388 kitchen)
     (at_kitchen_bread bread40)
     (at_kitchen_content content18)
     (not_allergic_gluten child236)
     (waiting child236 table276)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child236)
    )
  )
)
