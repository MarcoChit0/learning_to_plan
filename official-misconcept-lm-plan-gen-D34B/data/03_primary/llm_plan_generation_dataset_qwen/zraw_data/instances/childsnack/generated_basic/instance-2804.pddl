; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 600379

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child448 child221 - child
    bread258 bread189 - bread-portion
    content144 content400 - content-portion
    tray246 tray160 - tray
    table225 table331 table336 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray246 kitchen)
     (at tray160 kitchen)
     (at_kitchen_bread bread258)
     (at_kitchen_bread bread189)
     (at_kitchen_content content144)
     (at_kitchen_content content400)
     (not_allergic_gluten child221)
     (not_allergic_gluten child448)
     (waiting child448 table336)
     (waiting child221 table331)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child448)
     (served child221)
    )
  )
)
