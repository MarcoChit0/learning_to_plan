; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 409007

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child329 child469 - child
    bread378 bread46 - bread-portion
    content183 content11 - content-portion
    tray53 tray7 tray493 tray289 - tray
    table61 table52 table145 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray53 kitchen)
     (at tray7 kitchen)
     (at tray493 kitchen)
     (at tray289 kitchen)
     (at_kitchen_bread bread378)
     (at_kitchen_bread bread46)
     (at_kitchen_content content183)
     (at_kitchen_content content11)
     (not_allergic_gluten child329)
     (not_allergic_gluten child469)
     (waiting child329 table61)
     (waiting child469 table52)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child329)
     (served child469)
    )
  )
)
