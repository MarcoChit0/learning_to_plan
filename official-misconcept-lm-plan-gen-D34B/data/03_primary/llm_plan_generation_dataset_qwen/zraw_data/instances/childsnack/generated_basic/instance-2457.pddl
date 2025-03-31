; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 614154

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child216 child419 - child
    bread30 bread55 - bread-portion
    content295 content249 - content-portion
    tray436 tray306 - tray
    table196 table464 table346 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray436 kitchen)
     (at tray306 kitchen)
     (at_kitchen_bread bread30)
     (at_kitchen_bread bread55)
     (at_kitchen_content content295)
     (at_kitchen_content content249)
     (not_allergic_gluten child419)
     (not_allergic_gluten child216)
     (waiting child216 table346)
     (waiting child419 table346)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child216)
     (served child419)
    )
  )
)
