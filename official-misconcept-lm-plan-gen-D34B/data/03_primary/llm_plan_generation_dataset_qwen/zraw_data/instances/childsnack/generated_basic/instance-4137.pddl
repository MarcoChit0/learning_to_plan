; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 270257

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child231 child189 - child
    bread385 bread324 - bread-portion
    content410 content110 - content-portion
    tray22 tray419 tray219 - tray
    table22 table91 table364 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray22 kitchen)
     (at tray419 kitchen)
     (at tray219 kitchen)
     (at_kitchen_bread bread385)
     (at_kitchen_bread bread324)
     (at_kitchen_content content410)
     (at_kitchen_content content110)
     (not_allergic_gluten child231)
     (not_allergic_gluten child189)
     (waiting child231 table22)
     (waiting child189 table91)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child231)
     (served child189)
    )
  )
)
