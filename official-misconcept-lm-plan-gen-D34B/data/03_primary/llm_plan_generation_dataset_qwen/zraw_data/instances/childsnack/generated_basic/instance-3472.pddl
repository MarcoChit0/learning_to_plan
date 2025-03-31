; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 498520

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child486 child223 - child
    bread219 bread209 - bread-portion
    content228 content386 - content-portion
    tray191 tray400 - tray
    table289 table391 table51 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray191 kitchen)
     (at tray400 kitchen)
     (at_kitchen_bread bread219)
     (at_kitchen_bread bread209)
     (at_kitchen_content content228)
     (at_kitchen_content content386)
     (no_gluten_bread bread209)
     (no_gluten_content content386)
     (allergic_gluten child223)
     (not_allergic_gluten child486)
     (waiting child486 table51)
     (waiting child223 table289)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child486)
     (served child223)
    )
  )
)
