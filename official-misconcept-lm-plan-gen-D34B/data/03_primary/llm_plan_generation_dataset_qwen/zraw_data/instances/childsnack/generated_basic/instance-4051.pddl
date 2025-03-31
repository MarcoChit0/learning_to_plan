; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 249693

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child296 child349 child425 - child
    bread37 bread342 bread229 - bread-portion
    content35 content75 content45 - content-portion
    tray246 - tray
    table307 table495 table87 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray246 kitchen)
     (at_kitchen_bread bread37)
     (at_kitchen_bread bread342)
     (at_kitchen_bread bread229)
     (at_kitchen_content content35)
     (at_kitchen_content content75)
     (at_kitchen_content content45)
     (not_allergic_gluten child425)
     (not_allergic_gluten child349)
     (not_allergic_gluten child296)
     (waiting child296 table87)
     (waiting child349 table495)
     (waiting child425 table87)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child296)
     (served child349)
     (served child425)
    )
  )
)
