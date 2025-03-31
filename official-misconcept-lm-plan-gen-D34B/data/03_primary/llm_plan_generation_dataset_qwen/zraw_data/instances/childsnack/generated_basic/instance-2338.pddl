; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 331914

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child223 - child
    bread126 - bread-portion
    content327 - content-portion
    tray55 - tray
    table239 table393 table433 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray55 kitchen)
     (at_kitchen_bread bread126)
     (at_kitchen_content content327)
     (not_allergic_gluten child223)
     (waiting child223 table239)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child223)
    )
  )
)
