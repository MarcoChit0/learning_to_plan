; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 422267

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child36 child345 - child
    bread40 bread495 - bread-portion
    content321 content164 - content-portion
    tray66 tray432 - tray
    table313 table281 table252 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray66 kitchen)
     (at tray432 kitchen)
     (at_kitchen_bread bread40)
     (at_kitchen_bread bread495)
     (at_kitchen_content content321)
     (at_kitchen_content content164)
     (not_allergic_gluten child36)
     (not_allergic_gluten child345)
     (waiting child36 table252)
     (waiting child345 table281)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child36)
     (served child345)
    )
  )
)
