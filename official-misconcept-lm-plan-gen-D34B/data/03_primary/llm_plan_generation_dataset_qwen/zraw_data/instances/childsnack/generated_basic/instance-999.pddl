; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 547327

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child194 child154 child456 - child
    bread270 bread172 bread8 - bread-portion
    content321 content144 content315 - content-portion
    tray220 - tray
    table356 table336 table273 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray220 kitchen)
     (at_kitchen_bread bread270)
     (at_kitchen_bread bread172)
     (at_kitchen_bread bread8)
     (at_kitchen_content content321)
     (at_kitchen_content content144)
     (at_kitchen_content content315)
     (not_allergic_gluten child456)
     (not_allergic_gluten child194)
     (not_allergic_gluten child154)
     (waiting child194 table356)
     (waiting child154 table273)
     (waiting child456 table336)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child194)
     (served child154)
     (served child456)
    )
  )
)
