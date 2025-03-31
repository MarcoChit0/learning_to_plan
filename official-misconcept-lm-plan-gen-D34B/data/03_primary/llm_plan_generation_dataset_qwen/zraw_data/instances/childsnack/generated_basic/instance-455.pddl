; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 139639

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child6 child246 child355 - child
    bread55 bread230 bread52 - bread-portion
    content213 content82 content322 - content-portion
    tray182 - tray
    table202 table454 table280 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray182 kitchen)
     (at_kitchen_bread bread55)
     (at_kitchen_bread bread230)
     (at_kitchen_bread bread52)
     (at_kitchen_content content213)
     (at_kitchen_content content82)
     (at_kitchen_content content322)
     (not_allergic_gluten child6)
     (not_allergic_gluten child246)
     (not_allergic_gluten child355)
     (waiting child6 table202)
     (waiting child246 table454)
     (waiting child355 table280)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child6)
     (served child246)
     (served child355)
    )
  )
)
