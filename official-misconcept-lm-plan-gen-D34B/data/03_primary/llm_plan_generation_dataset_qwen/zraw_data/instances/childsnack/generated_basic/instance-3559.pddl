; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 295846

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child92 child257 - child
    bread57 bread156 - bread-portion
    content184 content141 - content-portion
    tray173 tray210 - tray
    table156 table260 table150 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray173 kitchen)
     (at tray210 kitchen)
     (at_kitchen_bread bread57)
     (at_kitchen_bread bread156)
     (at_kitchen_content content184)
     (at_kitchen_content content141)
     (not_allergic_gluten child92)
     (not_allergic_gluten child257)
     (waiting child92 table150)
     (waiting child257 table260)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child92)
     (served child257)
    )
  )
)
