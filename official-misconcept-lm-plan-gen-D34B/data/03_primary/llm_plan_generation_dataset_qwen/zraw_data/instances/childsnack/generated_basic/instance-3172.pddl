; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 420394

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child121 child35 - child
    bread104 bread251 - bread-portion
    content152 content36 - content-portion
    tray380 tray224 - tray
    table382 table44 table100 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray380 kitchen)
     (at tray224 kitchen)
     (at_kitchen_bread bread104)
     (at_kitchen_bread bread251)
     (at_kitchen_content content152)
     (at_kitchen_content content36)
     (not_allergic_gluten child121)
     (not_allergic_gluten child35)
     (waiting child121 table44)
     (waiting child35 table100)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child121)
     (served child35)
    )
  )
)
