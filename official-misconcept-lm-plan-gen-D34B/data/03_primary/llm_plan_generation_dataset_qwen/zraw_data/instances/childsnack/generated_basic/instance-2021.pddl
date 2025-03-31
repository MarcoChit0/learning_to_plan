; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 976764

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child24 child219 - child
    bread23 bread250 - bread-portion
    content492 content431 - content-portion
    tray69 tray66 tray124 - tray
    table449 table463 table34 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray69 kitchen)
     (at tray66 kitchen)
     (at tray124 kitchen)
     (at_kitchen_bread bread23)
     (at_kitchen_bread bread250)
     (at_kitchen_content content492)
     (at_kitchen_content content431)
     (not_allergic_gluten child219)
     (not_allergic_gluten child24)
     (waiting child24 table449)
     (waiting child219 table449)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child24)
     (served child219)
    )
  )
)
