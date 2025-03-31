; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 578662

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child408 child489 - child
    bread28 bread255 - bread-portion
    content199 content378 - content-portion
    tray181 tray342 - tray
    table400 table110 table243 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray181 kitchen)
     (at tray342 kitchen)
     (at_kitchen_bread bread28)
     (at_kitchen_bread bread255)
     (at_kitchen_content content199)
     (at_kitchen_content content378)
     (no_gluten_bread bread28)
     (no_gluten_content content378)
     (allergic_gluten child408)
     (not_allergic_gluten child489)
     (waiting child408 table400)
     (waiting child489 table400)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child408)
     (served child489)
    )
  )
)
