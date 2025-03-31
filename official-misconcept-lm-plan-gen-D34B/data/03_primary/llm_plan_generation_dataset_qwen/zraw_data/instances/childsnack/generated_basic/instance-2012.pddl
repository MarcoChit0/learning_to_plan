; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 863546

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child481 child152 - child
    bread76 bread289 - bread-portion
    content468 content426 - content-portion
    tray351 tray350 tray92 - tray
    table88 table322 table277 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray351 kitchen)
     (at tray350 kitchen)
     (at tray92 kitchen)
     (at_kitchen_bread bread76)
     (at_kitchen_bread bread289)
     (at_kitchen_content content468)
     (at_kitchen_content content426)
     (not_allergic_gluten child152)
     (not_allergic_gluten child481)
     (waiting child481 table322)
     (waiting child152 table322)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child481)
     (served child152)
    )
  )
)
