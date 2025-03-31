; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 888199

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child202 child77 child493 - child
    bread112 bread115 bread223 - bread-portion
    content326 content294 content465 - content-portion
    tray462 tray492 tray322 - tray
    table146 table132 table186 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray462 kitchen)
     (at tray492 kitchen)
     (at tray322 kitchen)
     (at_kitchen_bread bread112)
     (at_kitchen_bread bread115)
     (at_kitchen_bread bread223)
     (at_kitchen_content content326)
     (at_kitchen_content content294)
     (at_kitchen_content content465)
     (not_allergic_gluten child493)
     (not_allergic_gluten child202)
     (not_allergic_gluten child77)
     (waiting child202 table146)
     (waiting child77 table146)
     (waiting child493 table146)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child202)
     (served child77)
     (served child493)
    )
  )
)
