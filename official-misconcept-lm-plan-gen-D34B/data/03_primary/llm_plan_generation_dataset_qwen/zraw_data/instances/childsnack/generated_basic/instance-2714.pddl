; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 78603

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child151 child462 child180 - child
    bread15 bread130 bread27 - bread-portion
    content307 content201 content358 - content-portion
    tray430 tray40 tray246 - tray
    table202 table10 table47 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray430 kitchen)
     (at tray40 kitchen)
     (at tray246 kitchen)
     (at_kitchen_bread bread15)
     (at_kitchen_bread bread130)
     (at_kitchen_bread bread27)
     (at_kitchen_content content307)
     (at_kitchen_content content201)
     (at_kitchen_content content358)
     (not_allergic_gluten child151)
     (not_allergic_gluten child462)
     (not_allergic_gluten child180)
     (waiting child151 table10)
     (waiting child462 table47)
     (waiting child180 table47)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child151)
     (served child462)
     (served child180)
    )
  )
)
