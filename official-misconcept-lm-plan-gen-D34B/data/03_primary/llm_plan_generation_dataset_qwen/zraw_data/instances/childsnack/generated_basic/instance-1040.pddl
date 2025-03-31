; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 357382

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child440 child191 - child
    bread9 bread228 - bread-portion
    content439 content478 - content-portion
    tray479 tray1 tray79 - tray
    table186 table465 table423 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray479 kitchen)
     (at tray1 kitchen)
     (at tray79 kitchen)
     (at_kitchen_bread bread9)
     (at_kitchen_bread bread228)
     (at_kitchen_content content439)
     (at_kitchen_content content478)
     (not_allergic_gluten child191)
     (not_allergic_gluten child440)
     (waiting child440 table465)
     (waiting child191 table423)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child440)
     (served child191)
    )
  )
)
