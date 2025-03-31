; child-snack task with 2 children and 0.51 gluten factor 
; constant factor of 1.9
; random seed: 839996

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child377 child290 - child
    bread22 bread404 - bread-portion
    content219 content432 - content-portion
    tray342 tray183 tray111 - tray
    table21 table203 table350 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray342 kitchen)
     (at tray183 kitchen)
     (at tray111 kitchen)
     (at_kitchen_bread bread22)
     (at_kitchen_bread bread404)
     (at_kitchen_content content219)
     (at_kitchen_content content432)
     (no_gluten_bread bread404)
     (no_gluten_content content219)
     (allergic_gluten child290)
     (not_allergic_gluten child377)
     (waiting child377 table350)
     (waiting child290 table21)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child377)
     (served child290)
    )
  )
)
