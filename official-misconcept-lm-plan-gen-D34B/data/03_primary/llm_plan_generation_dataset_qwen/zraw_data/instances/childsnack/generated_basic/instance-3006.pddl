; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 191495

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child35 child324 - child
    bread270 bread344 - bread-portion
    content458 content439 - content-portion
    tray354 tray270 tray459 tray259 - tray
    table214 table20 table142 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray354 kitchen)
     (at tray270 kitchen)
     (at tray459 kitchen)
     (at tray259 kitchen)
     (at_kitchen_bread bread270)
     (at_kitchen_bread bread344)
     (at_kitchen_content content458)
     (at_kitchen_content content439)
     (not_allergic_gluten child324)
     (not_allergic_gluten child35)
     (waiting child35 table20)
     (waiting child324 table214)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child35)
     (served child324)
    )
  )
)
