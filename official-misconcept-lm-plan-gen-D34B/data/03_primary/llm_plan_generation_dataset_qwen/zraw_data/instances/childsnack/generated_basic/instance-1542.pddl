; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 877234

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child253 child459 - child
    bread134 bread29 - bread-portion
    content450 content391 - content-portion
    tray393 tray98 tray374 tray311 - tray
    table419 table289 table495 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray393 kitchen)
     (at tray98 kitchen)
     (at tray374 kitchen)
     (at tray311 kitchen)
     (at_kitchen_bread bread134)
     (at_kitchen_bread bread29)
     (at_kitchen_content content450)
     (at_kitchen_content content391)
     (not_allergic_gluten child459)
     (not_allergic_gluten child253)
     (waiting child253 table495)
     (waiting child459 table495)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child253)
     (served child459)
    )
  )
)
