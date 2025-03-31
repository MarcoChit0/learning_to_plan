; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 976127

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child197 child411 - child
    bread95 bread11 - bread-portion
    content279 content66 - content-portion
    tray38 tray295 tray365 tray82 - tray
    table94 table435 table21 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray38 kitchen)
     (at tray295 kitchen)
     (at tray365 kitchen)
     (at tray82 kitchen)
     (at_kitchen_bread bread95)
     (at_kitchen_bread bread11)
     (at_kitchen_content content279)
     (at_kitchen_content content66)
     (not_allergic_gluten child197)
     (not_allergic_gluten child411)
     (waiting child197 table21)
     (waiting child411 table21)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child197)
     (served child411)
    )
  )
)
