; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 46496

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child305 child407 - child
    bread24 bread158 - bread-portion
    content324 content215 - content-portion
    tray55 tray209 tray102 - tray
    table99 table117 table275 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray55 kitchen)
     (at tray209 kitchen)
     (at tray102 kitchen)
     (at_kitchen_bread bread24)
     (at_kitchen_bread bread158)
     (at_kitchen_content content324)
     (at_kitchen_content content215)
     (not_allergic_gluten child305)
     (not_allergic_gluten child407)
     (waiting child305 table275)
     (waiting child407 table117)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child305)
     (served child407)
    )
  )
)
