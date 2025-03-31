; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 825784

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child218 child168 - child
    bread212 bread304 - bread-portion
    content38 content454 - content-portion
    tray126 tray185 tray405 tray318 - tray
    table114 table358 table499 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray126 kitchen)
     (at tray185 kitchen)
     (at tray405 kitchen)
     (at tray318 kitchen)
     (at_kitchen_bread bread212)
     (at_kitchen_bread bread304)
     (at_kitchen_content content38)
     (at_kitchen_content content454)
     (no_gluten_bread bread212)
     (no_gluten_content content38)
     (allergic_gluten child218)
     (not_allergic_gluten child168)
     (waiting child218 table114)
     (waiting child168 table358)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child218)
     (served child168)
    )
  )
)
