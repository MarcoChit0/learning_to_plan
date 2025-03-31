; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 606073

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child58 child11 - child
    bread237 bread274 - bread-portion
    content301 content299 - content-portion
    tray95 tray128 tray53 - tray
    table450 table26 table253 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray95 kitchen)
     (at tray128 kitchen)
     (at tray53 kitchen)
     (at_kitchen_bread bread237)
     (at_kitchen_bread bread274)
     (at_kitchen_content content301)
     (at_kitchen_content content299)
     (not_allergic_gluten child11)
     (not_allergic_gluten child58)
     (waiting child58 table253)
     (waiting child11 table26)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child58)
     (served child11)
    )
  )
)
