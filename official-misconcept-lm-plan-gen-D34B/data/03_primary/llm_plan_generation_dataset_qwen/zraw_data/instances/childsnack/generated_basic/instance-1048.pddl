; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 198388

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child175 child438 - child
    bread452 bread76 - bread-portion
    content469 content460 - content-portion
    tray335 tray485 tray29 - tray
    table388 table185 table459 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray335 kitchen)
     (at tray485 kitchen)
     (at tray29 kitchen)
     (at_kitchen_bread bread452)
     (at_kitchen_bread bread76)
     (at_kitchen_content content469)
     (at_kitchen_content content460)
     (not_allergic_gluten child438)
     (not_allergic_gluten child175)
     (waiting child175 table459)
     (waiting child438 table185)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child175)
     (served child438)
    )
  )
)
