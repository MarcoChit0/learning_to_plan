; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 208557

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child204 child111 - child
    bread326 bread154 - bread-portion
    content465 content158 - content-portion
    tray263 tray250 - tray
    table490 table111 table169 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray263 kitchen)
     (at tray250 kitchen)
     (at_kitchen_bread bread326)
     (at_kitchen_bread bread154)
     (at_kitchen_content content465)
     (at_kitchen_content content158)
     (no_gluten_bread bread326)
     (no_gluten_content content158)
     (allergic_gluten child204)
     (not_allergic_gluten child111)
     (waiting child204 table490)
     (waiting child111 table490)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child204)
     (served child111)
    )
  )
)
